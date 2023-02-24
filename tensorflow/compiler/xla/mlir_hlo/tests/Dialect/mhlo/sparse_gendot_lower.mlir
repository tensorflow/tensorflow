// RUN: mlir-hlo-opt %s \
// RUN: --verify-diagnostics \
// RUN: --mhlo-test-lower-general-dot --canonicalize | FileCheck %s

#SV  = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
#CSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>
#COO = #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton-nu", "singleton" ] }>

//
// Vector-vector gendot.
//
// CHECK-LABEL: func.func @sparse_vecvec(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<10xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<f64> {
// CHECK:         %[[DOT:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<10xf64, #sparse_tensor.encoding<{{{.*}}}>>, tensor<10xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<f64>
// CHECK:         return %[[DOT]] : tensor<f64>
// CHECK:       }
//
func.func @sparse_vecvec(%arg0: tensor<10xf64, #SV>,
                         %arg1: tensor<10xf64, #SV>) -> tensor<f64> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0],
                                      rhs_contracting_dimensions = [0]>,
     precision_config = [#mhlo<precision DEFAULT>,
                         #mhlo<precision DEFAULT>]}
    : (tensor<10xf64, #SV>,
       tensor<10xf64, #SV>) -> tensor<f64>
  return %0 : tensor<f64>
}

//
// Matrix-vector gendot.
//
// CHECK-LABEL: func.func @sparse_matvec(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<3x5xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<5xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<3xf64> {
// CHECK:         %[[DOT:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3x5xf64, #sparse_tensor.encoding<{{{.*}}}>>, tensor<5xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<3xf64>
// CHECK:        return %[[DOT]] : tensor<3xf64>
// CHECK:       }
//
func.func @sparse_matvec(%arg0: tensor<3x5xf64, #CSR>,
                         %arg1: tensor<5xf64, #SV>) -> tensor<3xf64> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
                                      rhs_contracting_dimensions = [0]>,
     precision_config = [#mhlo<precision DEFAULT>,
                         #mhlo<precision DEFAULT>]}
    : (tensor<3x5xf64, #CSR>,
       tensor<5xf64, #SV>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}

//
// Matrix-matrix gendot.
//
// Ensures both transpositions are folded away after
// lowering dot_general to a direct dot operation.
//
// CHECK-LABEL: func.func @sparse_matmat(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<16x32xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<32x64xf64>) -> tensor<16x64xf64> {
// CHECK:         %[[DOT:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<16x32xf64, #sparse_tensor.encoding<{{{.*}}}>>, tensor<32x64xf64>) -> tensor<16x64xf64>
// CHECK:        return %[[DOT]] : tensor<16x64xf64>
// CHECK:       }
//
func.func @sparse_matmat(%arg0: tensor<16x32xf64, #CSR>,
                         %arg1: tensor<32x64xf64>) -> tensor<16x64xf64> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
                                      rhs_contracting_dimensions = [0]>,
    precision_config = [#mhlo<precision DEFAULT>,
                        #mhlo<precision DEFAULT>]}
    : (tensor<16x32xf64, #CSR>,
       tensor<32x64xf64>) -> tensor<16x64xf64>
  return %0 : tensor<16x64xf64>
}

//
// Higher-order gendot.
//
// A situation that introduces reshape operations.
//
// TODO: propagate sparsity better!
//
// CHECK-LABEL: func.func @sparse_tensor(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<197x12x64xf32>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<12x64x768xf32, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<197x768xf32> {
// CHECK:         %[[R1:.*]] = mhlo.reshape %[[ARG0]] : (tensor<197x12x64xf32>) -> tensor<197x768xf32>
// CHECK:         %[[R2:.*]] = mhlo.reshape %[[ARG1]] : (tensor<12x64x768xf32, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<768x768xf32>
// CHECK:         %[[R:.*]] = "mhlo.dot"(%[[R1]], %[[R2]]) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
// CHECK:         return %[[R]] : tensor<197x768xf32>
func.func @sparse_tensor(%arg0: tensor<197x12x64xf32>,
                         %arg1: tensor<12x64x768xf32, #COO>) -> tensor<197x768xf32> {
   %0 = "mhlo.dot_general"(%arg0, %arg1)
       {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1, 2],
                                          rhs_contracting_dimensions = [0, 1]>,
	precision_config = [#mhlo<precision DEFAULT>,
	                    #mhlo<precision DEFAULT>]}
    : (tensor<197x12x64xf32>,
       tensor<12x64x768xf32, #COO>) -> tensor<197x768xf32>
  return %0 : tensor<197x768xf32>
}
