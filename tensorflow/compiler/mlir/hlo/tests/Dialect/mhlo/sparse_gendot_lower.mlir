// RUN: mlir-hlo-opt %s \
// RUN:   --verify-diagnostics \
// RUN:   --mhlo-test-lower-general-dot | FileCheck %s

#SV  = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
#CSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

//
// Ensures both transpositions are folded away after
// lowering dot_general to a direct dot operation.
//
// CHECK-LABEL: func.func @sparse_gendot(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<32x32xf64, #{{.*}}>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<32x32xf64>) -> tensor<32x32xf64> {
// CHECK:         %[[DOT:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]]) {precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]} : (tensor<32x32xf64, #sparse_tensor.encoding<{{{.*}}}>>, tensor<32x32xf64>) -> tensor<32x32xf64>
// CHECK:        return %[[DOT]] : tensor<32x32xf64>
// CHECK:       }
//
func.func @sparse_gendot(%arg0: tensor<32x32xf64, #CSR>,
                         %arg1: tensor<32x32xf64>) -> tensor<32x32xf64> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
                                      rhs_contracting_dimensions = [0]>,
    precision_config = [#mhlo<"precision DEFAULT">,
                        #mhlo<"precision DEFAULT">]}
    : (tensor<32x32xf64, #CSR>,
       tensor<32x32xf64>) -> tensor<32x32xf64>
  return %0 : tensor<32x32xf64>
}

//
// A situation that introduces reshape operations.
//
// CHECK-LABEL: func.func @sparse_gendot_reshape(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<3x5xf64>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<5xf64, #{{.*}}>) -> tensor<3xf64> {
// CHECK:         %[[TRA:.*]] = "mhlo.transpose"(%[[ARG1]]) {permutation = dense<0> : tensor<1xi64>} : (tensor<5xf64, #{{.*}}>) -> tensor<5xf64>
// CHECK:         %[[RS0:.*]] = "mhlo.reshape"(%[[TRA]]) : (tensor<5xf64>) -> tensor<5x1xf64>
// CHECK:         %[[DOT:.*]] = "mhlo.dot"(%[[ARG0]], %[[RS0]]) {precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]} : (tensor<3x5xf64>, tensor<5x1xf64>) -> tensor<3x1xf64>
// CHECK:         %[[RS1:.*]] = "mhlo.reshape"(%[[DOT]]) : (tensor<3x1xf64>) -> tensor<3xf64>
// CHECK:        return %[[RS1]] : tensor<3xf64>
// CHECK:       }
//
func.func @sparse_gendot_reshape(%arg0: tensor<3x5xf64>,
                                 %arg1: tensor<5xf64, #SV>) -> tensor<3xf64> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
                                      rhs_contracting_dimensions = [0]>,
     precision_config = [#mhlo<"precision DEFAULT">,
                         #mhlo<"precision DEFAULT">]}
    : (tensor<3x5xf64>, tensor<5xf64, #SV>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}
