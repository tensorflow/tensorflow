// RUN: kernel-gen-opt %s --final-bufferize | FileCheck %s

// CHECK-LABEL: @extract_element
// CHECK-SAME: (%[[ARG:.*]]: memref<?xf32>) -> f32
func @extract_element(%arg : tensor<?xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[RESULT:.*]] = load %[[ARG]][%[[C0]]]
  // CHECK: return %[[RESULT]]
  %c0 = constant 0 : index
  %result = extract_element %arg[%c0] : tensor<?xf32>
  return %result : f32
}

// CHECK-LABEL: @tensor_from_elements
// CHECK-SAME: (%[[A:.*]]: f32) -> memref<3xf32>
func @tensor_from_elements(%a : f32) -> tensor<3xf32> {
  // CHECK: %[[B:.*]] = constant 1.2
  // CHECK: %[[C:.*]] = constant 2.3
  // CHECK: %[[MEM:.*]] = alloc() : memref<3xf32>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: store %[[A]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: store %[[B]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: store %[[C]], %[[MEM]][%[[C2]]] : memref<3xf32>
  // CHECK: return %[[MEM]] : memref<3xf32>
  %b = constant 1.2 : f32
  %c = constant 2.3 : f32
  %result = tensor_from_elements %a, %b, %c : tensor<3xf32>
  return %result : tensor<3xf32>
}

// CHECK-LABEL: @dynamic_tensor_from_elements
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>) -> memref<?xindex>
func @dynamic_tensor_from_elements(%arg : tensor<*xf32>) -> tensor<?xindex> {
  // CHECK: %[[C3:.*]] = constant 3 : index
  // CHECK: %[[MEM:.*]] = alloca(%c3) : memref<?xindex>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[C3]]) step (%[[C1]]) {
  // CHECK:   %[[ELEM:.*]] = dim %[[ARG]], %[[I]] : memref<*xf32>
  // CHECK:   store %[[ELEM]], %[[MEM]][%[[I]]] : memref<?xindex>
  // CHECK:   scf.yield
  // CHECK: }
  // CHECK: return %[[MEM]] : memref<?xindex>
  %c3 = constant 3 : index
  %result = dynamic_tensor_from_elements %c3 {
  ^bb0(%i : index):
    %elem = dim %arg, %i : tensor<*xf32>
    yield %elem : index
  } : tensor<?xindex>
  return %result : tensor<?xindex>
}

// CHECK-LABEL: @assuming
// CHECK-SAME: (%[[WITNESS:.*]]: !shape.witness, %[[ARG:.*]]: memref<?xf32>)
// CHECK-SAME: -> memref<?xf32>
func @assuming(%witness: !shape.witness, %arg : memref<?xf32>)
              -> tensor<?xf32> {
  // CHECK-NEXT: %[[ASSUMING_RESULT:.*]] = shape.assuming %[[WITNESS]]
  // CHECK-SAME:     -> (memref<?xf32>) {
  // CHECK-NEXT:   shape.assuming_yield %[[ARG]] : memref<?xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[ASSUMING_RESULT]] : memref<?xf32>
  %assuming_result = shape.assuming %witness -> (tensor<?xf32>) {
    %result = tensor_load %arg : memref<?xf32>
    shape.assuming_yield %result : tensor<?xf32>
  }
  return %assuming_result : tensor<?xf32>
}

// CHECK-LABEL: @const
// CHECK-SAME: -> memref<3xf32>
func @const() -> tensor<3xf32> {
  // CHECK: %[[MEM:.*]] = alloca() : memref<3xf32>
  // CHECK: %[[C4:.*]] = constant 4.000000e+00 : f32
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: %[[C5:.*]] = constant 5.000000e+00 : f32
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: store %[[C5]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: %[[C6:.*]] = constant 6.000000e+00 : f32
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: store %[[C6]], %[[MEM]][%[[C2]]] : memref<3xf32>
  // CHECK-NEXT: return %[[MEM]] : memref<3xf32>
  %result = constant dense<[4.0, 5.0, 6.0]> : tensor<3xf32>
  return %result : tensor<3xf32>
}

// CHECK-LABEL: @const_splat
// CHECK-SAME: -> memref<3xf32>
func @const_splat() -> tensor<3xf32> {
  // CHECK: %[[MEM:.*]] = alloca() : memref<3xf32>
  // CHECK: %[[C4:.*]] = constant 4.000000e+00 : f32
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C2]]] : memref<3xf32>
  // CHECK-NEXT: return %[[MEM]] : memref<3xf32>
  %result = constant dense<4.0> : tensor<3xf32>
  return %result : tensor<3xf32>
}
