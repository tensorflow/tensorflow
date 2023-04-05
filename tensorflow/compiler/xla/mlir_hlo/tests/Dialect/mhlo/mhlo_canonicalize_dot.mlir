// RUN: mlir-hlo-opt %s --split-input-file --hlo-canonicalize-dot | FileCheck %s


func.func @dot(%lhs: tensor<1x300xf32>, %rhs: tensor<300x1xf32>)
               -> tensor<1x1xf32> {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<1x300xf32>, tensor<300x1xf32>)
                                 -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: func.func @dot(
// CHECK:       mhlo.dot{{.*}}(tensor<300xf32>, tensor<300xf32>) -> tensor<f32>

// -----

func.func @matvec(%lhs: tensor<18x300xf32>, %rhs: tensor<300x1xf32>)
                  -> tensor<18x1xf32> {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<18x300xf32>, tensor<300x1xf32>)
                                 -> tensor<18x1xf32>
  return %0 : tensor<18x1xf32>
}

// CHECK-LABEL: func.func @matvec(
// CHECK:       mhlo.dot{{.*}}(tensor<18x300xf32>, tensor<300xf32>) -> tensor<18xf32>

// -----

func.func @matmul(%lhs: tensor<18x1xf32>, %rhs: tensor<1x300xf32>)
                  -> tensor<18x300xf32> {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<18x1xf32>, tensor<1x300xf32>)
                                 -> tensor<18x300xf32>
  return %0 : tensor<18x300xf32>
}

// CHECK-LABEL: func.func @matmul(
// CHECK:       mhlo.dot{{.*}}(tensor<18x1xf32>, tensor<1x300xf32>) -> tensor<18x300xf32>

// -----

func.func @vecmat(%lhs: tensor<1x300xf32>, %rhs: tensor<300x300xf32>)
                  -> tensor<1x300xf32> {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<1x300xf32>, tensor<300x300xf32>)
                                 -> tensor<1x300xf32>
  return %0 : tensor<1x300xf32>
}

// CHECK-LABEL: func.func @vecmat(
// CHECK:       mhlo.dot{{.*}}(tensor<300xf32>, tensor<300x300xf32>) -> tensor<300xf32>

// -----

func.func @unit_vecmat(%lhs: tensor<1x1xf32>, %rhs: tensor<1x300xf32>)
                       -> tensor<1x300xf32> {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<1x1xf32>, tensor<1x300xf32>)
                                 -> tensor<1x300xf32>
  return %0 : tensor<1x300xf32>
}

// CHECK-LABEL: func.func @unit_vecmat(
// CHECK:       mhlo.dot{{.*}}(tensor<1xf32>, tensor<1x300xf32>) -> tensor<300xf32>

// -----

func.func @unit_dot(%lhs: tensor<1x1xf32>, %rhs: tensor<1x1xf32>)
                    -> tensor<1x1xf32> {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<1x1xf32>, tensor<1x1xf32>)
                                 -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: func.func @unit_dot(
// CHECK:       mhlo.dot{{.*}}(tensor<1xf32>, tensor<1xf32>) -> tensor<f32>

// -----

func.func @dyn_vecmat(%lhs: tensor<1x?xf32>, %rhs: tensor<?x300xf32>)
                      -> tensor<1x300xf32> {
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<1x?xf32>, tensor<?x300xf32>)
                                 -> tensor<1x300xf32>
  return %0 : tensor<1x300xf32>
}

// CHECK-LABEL: func.func @dyn_vecmat(
// CHECK:       mhlo.dot{{.*}}(tensor<?xf32>, tensor<?x300xf32>) -> tensor<300xf32>
