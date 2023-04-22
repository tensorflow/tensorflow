// RUN: kernel-gen-opt -split-input-file -kernelgen-lower-index-cast %s | FileCheck %s

// index_cast of static tensor
// CHECK-LABEL: func @f
func @f(%arg0 : tensor<10xi32>) -> tensor<10xindex> {
  // CHECK: %[[TENSOR:.*]] = tensor.generate {
  // CHECK: ^bb0(%arg1: index):
  // CHECK:   %[[E:.*]] = tensor.extract %arg0[%arg1] : tensor<10xi32>
  // CHECK:   %[[C:.*]] = index_cast %[[E]] : i32 to index
  // CHECK:   tensor.yield %[[C]] : index
  // CHECK: } : tensor<10xindex>
  // CHECK: return %[[TENSOR]] : tensor<10xindex>
  %0 = index_cast %arg0 : tensor<10xi32> to tensor<10xindex>
  return %0 : tensor<10xindex>
}

// -----

// index_cast of dynamic tensor
func @f(%arg0 : tensor<?xi32>) -> tensor<?xindex> {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?xi32>
  // CHECK: %[[TENSOR:.*]] = tensor.generate %[[DIM]] {
  // CHECK: ^bb0(%arg1: index):
  // CHECK:   %[[E:.*]] = tensor.extract %arg0[%arg1] : tensor<?xi32>
  // CHECK:   %[[C:.*]] = index_cast %[[E]] : i32 to index
  // CHECK:   tensor.yield %[[C]] : index
  // CHECK: } : tensor<?xindex>
  // CHECK: return %[[TENSOR]] : tensor<?xindex>
  %0 = index_cast %arg0 : tensor<?xi32> to tensor<?xindex>
  return %0 : tensor<?xindex>
}
