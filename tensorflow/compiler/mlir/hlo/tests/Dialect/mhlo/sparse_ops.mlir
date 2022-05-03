// RUN: mlir-hlo-opt %s -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// Tests for sparse types. Note that most dense MHLO ops can be made sparse
// by simply annotating one or more of the tensor types as sparse. Other than
// subtle printing and parsing difference (due to having different input and
// output types), dense or sparse ops are semantically equivalent.

#SV = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

//
// Dense unary and binary eltwise.
//

// CHECK-LABEL: func @dense_abs_eltwise(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32>)
//       CHECK: %[[T:.*]] = mhlo.abs %[[A]] : tensor<10x20xf32>
//       CHECK: return %[[T]] : tensor<10x20xf32>
func.func @dense_abs_eltwise(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = mhlo.abs %arg0 : tensor<10x20xf32>
  func.return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @dense_add_eltwise(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32>,
//  CHECK-SAME: %[[B:.*]]: tensor<10x20xf32>)
//       CHECK: %[[T:.*]] = mhlo.add %[[A]], %[[B]] : tensor<10x20xf32>
//       CHECK: return %[[T]] : tensor<10x20xf32>
func.func @dense_add_eltwise(%arg0: tensor<10x20xf32>,
                        %arg1: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<10x20xf32>
  func.return %0 : tensor<10x20xf32>
}

//
// Sparse unary eltwise.
//

// CHECK-LABEL: func @sparse_abs_eltwise1(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32, #{{.*}}>)
//       CHECK: %[[T:.*]] = mhlo.abs(%[[A]]) : (tensor<10x20xf32, #{{.*}}>) -> tensor<10x20xf32>
//       CHECK: return %[[T]] : tensor<10x20xf32>
func.func @sparse_abs_eltwise1(%arg0: tensor<10x20xf32, #CSR>) -> tensor<10x20xf32> {
  %0 = mhlo.abs(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32>
  func.return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @sparse_abs_eltwise2(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32, #{{.*}}>)
//       CHECK: %[[T:.*]] = mhlo.abs %[[A]] : tensor<10x20xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<10x20xf32, #{{.*}}>
func.func @sparse_abs_eltwise2(%arg0: tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #CSR> {
  %0 = mhlo.abs %arg0  : tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}

// CHECK-LABEL: func @sparse_abs_eltwise3(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32, #{{.*}}>)
//       CHECK: %[[T:.*]] = mhlo.abs(%[[A]]) : (tensor<10x20xf32, #{{.*}}>) -> tensor<10x20xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<10x20xf32, #{{.*}}>
func.func @sparse_abs_eltwise3(%arg0: tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #DCSR> {
  %0 = mhlo.abs(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #DCSR>
  func.return %0 : tensor<10x20xf32, #DCSR>
}

// CHECK-LABEL: func @sparse_abs_eltwise4(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32>)
//       CHECK: %[[T:.*]] = mhlo.abs(%[[A]]) : (tensor<10x20xf32>) -> tensor<10x20xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<10x20xf32, #{{.*}}>
func.func @sparse_abs_eltwise4(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32, #CSR> {
  %0 = mhlo.abs(%arg0) : (tensor<10x20xf32>) -> tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}

//
// Sparse binary eltwise.
//

// CHECK-LABEL: func @sparse_add_eltwise1(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32, #{{.*}}>,
//  CHECK-SAME: %[[B:.*]]: tensor<10x20xf32>)
//       CHECK: %[[T:.*]] = mhlo.add(%[[A]], %[[B]]) : (tensor<10x20xf32, #{{.*}}>, tensor<10x20xf32>) -> tensor<10x20xf32>
//       CHECK: return %[[T]] : tensor<10x20xf32>
func.func @sparse_add_eltwise1(%arg0: tensor<10x20xf32, #CSR>,
                               %arg1: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = mhlo.add(%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                 tensor<10x20xf32>) -> tensor<10x20xf32>
  func.return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @sparse_add_eltwise2(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32, #{{.*}}>,
//  CHECK-SAME: %[[B:.*]]: tensor<10x20xf32, #{{.*}}>)
//       CHECK: %[[T:.*]] = mhlo.add(%[[A]], %[[B]]) : (tensor<10x20xf32, #{{.*}}>, tensor<10x20xf32, #{{.*}}>) -> tensor<10x20xf32>
//       CHECK: return %[[T]] : tensor<10x20xf32>
func.func @sparse_add_eltwise2(%arg0: tensor<10x20xf32, #CSR>,
                               %arg1: tensor<10x20xf32, #DCSR>)
                                   -> tensor<10x20xf32> {
  %0 = mhlo.add(%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                 tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32>
  func.return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @sparse_add_eltwise3(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32, #{{.*}}>,
//  CHECK-SAME: %[[B:.*]]: tensor<10x20xf32, #{{.*}}>)
//       CHECK: %[[T:.*]] = mhlo.add(%[[A]], %[[B]]) : (tensor<10x20xf32, #{{.*}}>, tensor<10x20xf32, #{{.*}}>) -> tensor<10x20xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<10x20xf32, #{{.*}}>
func.func @sparse_add_eltwise3(%arg0: tensor<10x20xf32, #CSR>,
                               %arg1: tensor<10x20xf32, #DCSR>)
                                   -> tensor<10x20xf32, #CSR> {
  %0 = mhlo.add(%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                 tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}

// CHECK-LABEL: func @sparse_add_eltwise4(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32>,
//  CHECK-SAME: %[[B:.*]]: tensor<10x20xf32>)
//       CHECK: %[[T:.*]] = mhlo.add(%[[A]], %[[B]]) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<10x20xf32, #{{.*}}>
func.func @sparse_add_eltwise4(%arg0: tensor<10x20xf32>,
                               %arg1: tensor<10x20xf32>)
                                   -> tensor<10x20xf32, #CSR> {
  %0 = mhlo.add(%arg0, %arg1) : (tensor<10x20xf32>,
                                 tensor<10x20xf32>) -> tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}

// CHECK-LABEL: func @sparse_add_eltwise5(
//  CHECK-SAME: %[[A:.*]]: tensor<10x20xf32, #{{.*}}>,
//  CHECK-SAME: %[[B:.*]]: tensor<10x20xf32, #{{.*}}>)
//       CHECK: %[[T:.*]] = mhlo.add %[[A]], %[[B]] : tensor<10x20xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<10x20xf32, #{{.*}}>
func.func @sparse_add_eltwise5(%arg0: tensor<10x20xf32, #CSR>,
                               %arg1: tensor<10x20xf32, #CSR>)
                                   -> tensor<10x20xf32, #CSR> {
  %0 = mhlo.add %arg0, %arg1 : tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}

// CHECK-LABEL: func @quantization_and_sparse(
//  CHECK-SAME: %[[A:.*]]: tensor<1x!quant.uniform<i8:f32, 1.000000e+00:17>, #{{.*}}>)
//       CHECK: return %[[A]] : tensor<1x!quant.uniform<i8:f32, 1.000000e+00:17>, #{{.*}}>
func.func @quantization_and_sparse(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>, #SV>)
                                       -> tensor<1x!quant.uniform<i8:f32, 1.0:17>, #SV> {
  return %arg0 : tensor<1x!quant.uniform<i8:f32, 1.0:17>, #SV>
}
