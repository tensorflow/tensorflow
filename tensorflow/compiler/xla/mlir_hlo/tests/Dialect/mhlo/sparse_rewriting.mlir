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
// CHECK-SAME:    %[[ARG0:.*]]: tensor<100xf64>) -> tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[VAL:.*]] = mhlo.abs %[[ARG0]] : (tensor<100xf64>) -> tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @rewrite_unary(%arg0: tensor<100xf64>) -> tensor<100xf64, #SV> {
  %0 = mhlo.abs %arg0 : tensor<100xf64>
  %1 = sparse_tensor.convert %0 : tensor<100xf64> to tensor<100xf64, #SV>
  return %1 : tensor<100xf64, #SV>
}

// CHECK-LABEL: func @rewrite_binary(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<100xf64>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[VAL:.*]] = mhlo.multiply %[[ARG0]], %[[ARG1]] : (tensor<100xf64>, tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @rewrite_binary(%arg0: tensor<100xf64>,
                          %arg1: tensor<100xf64, #SV>) -> tensor<100xf64, #SV> {
  %0 = mhlo.multiply %arg0, %arg1 : (tensor<100xf64>, tensor<100xf64, #SV>) -> tensor<100xf64>
  %1 = sparse_tensor.convert %0 : tensor<100xf64> to tensor<100xf64, #SV>
  return %1 : tensor<100xf64, #SV>
}

// CHECK-LABEL: func @rewrite_binary_override(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[VAL:.*]] = mhlo.multiply %[[ARG0]], %[[ARG1]] : (tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @rewrite_binary_override(%arg0: tensor<10x10xf64, #CSR>,
                                   %arg1: tensor<10x10xf64, #CSR>) -> tensor<10x10xf64, #DCSR> {
  %0 = mhlo.multiply %arg0, %arg1 : (tensor<10x10xf64, #CSR>, tensor<10x10xf64, #CSR>) -> tensor<10x10xf64, #CSR>
  %1 = sparse_tensor.convert %0 : tensor<10x10xf64, #CSR> to tensor<10x10xf64, #DCSR>
  return %1 : tensor<10x10xf64, #DCSR>
}

// CHECK-LABEL: func @rewrite_convert(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x10xf64>) -> tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[VAL:.*]] = sparse_tensor.convert %[[ARG0]] : tensor<10x10xf64> to tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
func.func @rewrite_convert(%arg0: tensor<10x10xf64>) -> tensor<10x10xf64, #CSR> {
  %0 = sparse_tensor.convert %arg0 : tensor<10x10xf64> to tensor<10x10xf64, #DCSR>
  %1 = sparse_tensor.convert %0 : tensor<10x10xf64, #DCSR> to tensor<10x10xf64, #CSR>
  %2 = sparse_tensor.convert %1 : tensor<10x10xf64, #CSR> to tensor<10x10xf64, #CSR>
  return %2 : tensor<10x10xf64, #CSR>
}

// CHECK-LABEL: func @rewrite_convert_nop(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
// CHECK-NEXT:    %[[RES:.*]] = sparse_tensor.convert %[[ARG0]]
// CHECK-NEXT:    return %[[RES]] : tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
func.func @rewrite_convert_nop(%arg0: tensor<10x10xf64, #CSR>) -> tensor<10x10xf64, #CSR> {
  %0 = sparse_tensor.convert %arg0 : tensor<10x10xf64, #CSR> to tensor<10x10xf64, #DCSR>
  %1 = sparse_tensor.convert %0 : tensor<10x10xf64, #DCSR> to tensor<10x10xf64, #CSR>
  %2 = sparse_tensor.convert %1 : tensor<10x10xf64, #CSR> to tensor<10x10xf64, #CSR>
  return %2 : tensor<10x10xf64, #CSR>
}

// CHECK-LABEL: func @rewrite_transpose(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<100x200xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<200x100xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[VAL:.*]] = "mhlo.transpose"(%[[ARG0]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<100x200xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK-NEXT:    return %[[VAL:.*]] : tensor<200x100xf64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @rewrite_transpose(%arg0: tensor<100x200xf64, #CSR>) -> tensor<200x100xf64, #CSR> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<100x200xf64, #CSR>) -> tensor<200x100xf64>
  %1 = sparse_tensor.convert %0 : tensor<200x100xf64> to tensor<200x100xf64, #CSR>
  return %1 : tensor<200x100xf64, #CSR>
}

// CHECK-LABEL: func.func @rewrite_dot(
// CHECK-SAME:    %[[ARG0:.*0]]: tensor<5x5xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:    %[[ARG1:.*1]]: tensor<5x5xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<5x5xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[VAL:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]])
// CHECK:         return %[[VAL]] : tensor<5x5xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
func.func @rewrite_dot(%arg0: tensor<5x5xf64, #CSR>,
                       %arg1: tensor<5x5xf64, #CSR>) -> tensor<5x5xf64, #CSR> {
  %0 = "mhlo.dot"(%arg0, %arg1)
      {precision_config = [#mhlo<precision DEFAULT>,
                          #mhlo<precision DEFAULT>]}
     : (tensor<5x5xf64, #CSR>,
        tensor<5x5xf64, #CSR>) -> tensor<5x5xf64>
  %1 = sparse_tensor.convert %0 : tensor<5x5xf64> to tensor<5x5xf64, #CSR>
  return %1 : tensor<5x5xf64, #CSR>
}

// CHECK-LABEL: func.func @rewrite_general_dot(
// CHECK-SAME:    %[[ARG0:.*0]]: tensor<5x5xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:    %[[ARG1:.*1]]: tensor<5x5xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<5x5xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[VAL:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]])
// CHECK:         return %[[VAL]] : tensor<5x5xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
func.func @rewrite_general_dot(%arg0: tensor<5x5xf64, #CSR>,
                               %arg1: tensor<5x5xf64, #CSR>) -> tensor<5x5xf64, #CSR> {
   %0 = "mhlo.dot_general"(%arg0, %arg1)
       {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
                                          rhs_contracting_dimensions = [0]>,
	precision_config = [#mhlo<precision DEFAULT>,
	                    #mhlo<precision DEFAULT>]}
     : (tensor<5x5xf64, #CSR>,
        tensor<5x5xf64, #CSR>) -> tensor<5x5xf64>
  %1 = sparse_tensor.convert %0 : tensor<5x5xf64> to tensor<5x5xf64, #CSR>
  return %1 : tensor<5x5xf64, #CSR>
}

// CHECK-LABEL:  func.func @rewrite_elt_convert(
// CHECK-SAME:     %[[ARG0:.*0]]: tensor<5x5xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<5x5xf32, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:          %[[VAL:.*]] = sparse_tensor.convert %[[ARG0]]
// CHECK:          return %[[VAL]] : tensor<5x5xf32, #sparse_tensor.encoding<{{{.*}}}>>
func.func @rewrite_elt_convert(%arg0: tensor<5x5xf64, #CSR>) -> tensor<5x5xf32, #CSR> {
  %0 = "mhlo.convert"(%arg0) : (tensor<5x5xf64, #CSR>) -> tensor<5x5xf32, #CSR>
  return %0 : tensor<5x5xf32, #CSR>
}

// CHECK-LABEL:  func.func @concatenate_sparse(
// CHECK-SAME:     %[[ARG0:.*0]]: tensor<100x100xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:     %[[ARG1:.*1]]: tensor<100x100xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<200x100xf64, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:          %[[VAL:.*]] = sparse_tensor.concatenate %[[ARG0]], %[[ARG1]] {dimension = 0
// CHECK:          return %[[VAL]] : tensor<200x100xf64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @concatenate_sparse(%arg0: tensor<100x100xf64, #CSR>, %arg1: tensor<100x100xf64, #CSR>) -> tensor<200x100xf64, #CSR> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<100x100xf64, #CSR>, tensor<100x100xf64, #CSR>) -> tensor<200x100xf64, #CSR>
  return %0 : tensor<200x100xf64, #CSR>
}
