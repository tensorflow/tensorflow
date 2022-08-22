// RUN: mlir-hlo-opt %s \
// RUN:   --verify-diagnostics \
// RUN:   --mhlo-sparse-rewriting  | FileCheck %s

// Verifies that mhlo sparse tensor type rewriting occurs.

#SV= #sparse_tensor.encoding<{ dimLevelType = ["compressed"] }>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

// CHECK-LABEL: func @rewrite_unary(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<100xf64>) -> tensor<100xf64, #{{.*}}> {
// CHECK:         %[[VAL:.*]] = mhlo.abs(%[[ARG0]]) : (tensor<100xf64>) -> tensor<100xf64, #{{.*}}>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<100xf64, #{{.*}}>
func.func @rewrite_unary(%arg0: tensor<100xf64>) -> tensor<100xf64, #SV> {
  %0 = mhlo.abs %arg0 : tensor<100xf64>
  %1 = sparse_tensor.convert %0 : tensor<100xf64> to tensor<100xf64, #SV>
  return %1 : tensor<100xf64, #SV>
}

// CHECK-LABEL: func @rewrite_binary(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<100xf64>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<100xf64, #{{.*}}>) -> tensor<100xf64, #{{.*}}> {
// CHECK:         %[[VAL:.*]] = mhlo.multiply(%[[ARG0]], %[[ARG1]]) : (tensor<100xf64>, tensor<100xf64, #{{.*}}>) -> tensor<100xf64, #{{.*}}>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<100xf64, #{{.*}}>
func.func @rewrite_binary(%arg0: tensor<100xf64>,
                          %arg1: tensor<100xf64, #SV>) -> tensor<100xf64, #SV> {
  %0 = mhlo.multiply(%arg0, %arg1) : (tensor<100xf64>, tensor<100xf64, #SV>) -> tensor<100xf64>
  %1 = sparse_tensor.convert %0 : tensor<100xf64> to tensor<100xf64, #SV>
  return %1 : tensor<100xf64, #SV>
}

// CHECK-LABEL: func @rewrite_binary_override(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x10xf64, #{{.*}}>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<10x10xf64, #{{.*}}>) -> tensor<10x10xf64, #{{.*}}> {
// CHECK:         %[[VAL:.*]] = mhlo.multiply(%[[ARG0]], %[[ARG1]]) : (tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<10x10xf64, #{{.*}}>
func.func @rewrite_binary_override(%arg0: tensor<10x10xf64, #CSR>,
                                   %arg1: tensor<10x10xf64, #CSR>) -> tensor<10x10xf64, #DCSR> {
  %0 = mhlo.multiply(%arg0, %arg1) : (tensor<10x10xf64, #CSR>, tensor<10x10xf64, #CSR>) -> tensor<10x10xf64, #CSR>
  %1 = sparse_tensor.convert %0 : tensor<10x10xf64, #CSR> to tensor<10x10xf64, #DCSR>
  return %1 : tensor<10x10xf64, #DCSR>
}

// CHECK-LABEL: func @rewrite_convert(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x10xf64>) -> tensor<10x10xf64, #{{.*}}> {
// CHECK:         %[[VAL:.*]] = sparse_tensor.convert %[[ARG0]] : tensor<10x10xf64> to tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
func.func @rewrite_convert(%arg0: tensor<10x10xf64>) -> tensor<10x10xf64, #CSR> {
  %0 = sparse_tensor.convert %arg0 : tensor<10x10xf64> to tensor<10x10xf64, #DCSR>
  %1 = sparse_tensor.convert %0 : tensor<10x10xf64, #DCSR> to tensor<10x10xf64, #CSR>
  %2 = sparse_tensor.convert %1 : tensor<10x10xf64, #CSR> to tensor<10x10xf64, #CSR>
  return %2 : tensor<10x10xf64, #CSR>
}

// CHECK-LABEL: func @rewrite_convert_nop(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x10xf64, #{{.*}}>) -> tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
// CHECK-NEXT:    return %[[ARG0:.*]] : tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
func.func @rewrite_convert_nop(%arg0: tensor<10x10xf64, #CSR>) -> tensor<10x10xf64, #CSR> {
  %0 = sparse_tensor.convert %arg0 : tensor<10x10xf64, #CSR> to tensor<10x10xf64, #DCSR>
  %1 = sparse_tensor.convert %0 : tensor<10x10xf64, #DCSR> to tensor<10x10xf64, #CSR>
  %2 = sparse_tensor.convert %1 : tensor<10x10xf64, #CSR> to tensor<10x10xf64, #CSR>
  return %2 : tensor<10x10xf64, #CSR>
}

// CHECK-LABEL: func @rewrite_transpose(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<100x200xf64, #{{.*}}>) -> tensor<200x100xf64, #{{.*}}> {
// CHECK:         %[[VAL:.*]] = "mhlo.transpose"(%[[ARG0]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<100x200xf64, #{{.*}}>) -> tensor<200x100xf64, #{{.*}}>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<200x100xf64, #{{.*}}>
func.func @rewrite_transpose(%arg0: tensor<100x200xf64, #CSR>) -> tensor<200x100xf64, #CSR> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<100x200xf64, #CSR>) -> tensor<200x100xf64>
  %1 = sparse_tensor.convert %0 : tensor<200x100xf64> to tensor<200x100xf64, #CSR>
  return %1 : tensor<200x100xf64, #CSR>
}

// CHECK-LABEL:  func.func @concatenate_sparse(
// CHECK-SAME:     %[[TMP_arg0:.*0]]: tensor<100x100xf64,
// CHECK-SAME:     %[[TMP_arg1:.*1]]: tensor<100x100xf64,
// CHECK-SAME:     -> tensor<200x100xf64,
// CHECK:          %[[TMP_0:.*]] = sparse_tensor.concatenate %[[TMP_arg0]], %[[TMP_arg1]] {dimension = 0
// CHECK:          return %[[TMP_0]] : tensor<200x100xf64,
func.func @concatenate_sparse(%arg0: tensor<100x100xf64, #CSR>, %arg1: tensor<100x100xf64, #CSR>) -> tensor<200x100xf64, #CSR> attributes {llvm.emit_c_interface} {
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<100x100xf64, #CSR>, tensor<100x100xf64, #CSR>) -> tensor<200x100xf64, #CSR>
  return %0 : tensor<200x100xf64, #CSR>
}
