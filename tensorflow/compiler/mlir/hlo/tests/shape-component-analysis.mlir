// RUN: mlir-hlo-opt --test-print-shape-components --split-input-file %s | FileCheck %s

// CHECK-LABEL: Testing : assuming
func @assuming(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : !shape.witness) -> tensor<2xi32> {
  %0:2 = shape.assuming %arg2 -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    shape.assuming_yield %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>
  }
  %1 = shape.shape_of %0#0 : tensor<?x?xf32> -> tensor<2xindex>
  %2 = shape.shape_of %0#1 : tensor<?x?xf32> -> tensor<2xindex>
  %3 = arith.index_cast %1 : tensor<2xindex> to tensor<2xi32>
  %4 = arith.index_cast %2 : tensor<2xindex> to tensor<2xi32>
  // CHECK:      Value info for %5 = mhlo.add %3, %4 : tensor<2xi32>
  // CHECK-NEXT:   s0 + s1 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  // CHECK-NEXT:     s1 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[0]
  // CHECK-NEXT:   s0 + s1 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  // CHECK-NEXT:     s1 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[1]
  %5 = mhlo.add %3, %4 : tensor<2xi32>
  // CHECK:      Value info for %6 = mhlo.multiply %5, %4 : tensor<2xi32>
  // CHECK-NEXT:   (s0 + s1) * s2 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  // CHECK-NEXT:     s1 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[0]
  // CHECK-NEXT:     s2 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[0]
  // CHECK-NEXT:   (s0 + s1) * s2 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  // CHECK-NEXT:     s1 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[1]
  // CHECK-NEXT:     s2 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[1]
  %6 = mhlo.multiply %5, %4 : tensor<2xi32>
  return %6 : tensor<2xi32>
}

// -----

// CHECK-LABEL: Testing : num_elements
func @num_elements(%arg0: tensor<?x8x?x64xf32>) -> index {
  // CHECK:      Value info for %0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x8x?x64xf32>' at index: 0)[0]
  // CHECK-NEXT:   8
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x8x?x64xf32>' at index: 0)[2]
  // CHECK-NEXT:   64
  %0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  // CHECK:      Value info for %1 = shape.num_elements %0 : tensor<4xindex> -> index:
  // CHECK-NEXT:   (s0 * s1) * 512 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x8x?x64xf32>' at index: 0)[0]
  // CHECK-NEXT:     s1 = shapeof(<block argument> of type 'tensor<?x8x?x64xf32>' at index: 0)[2]
  %1 = shape.num_elements %0 : tensor<4xindex> -> index
  return %1 : index
}

// -----

// CHECK-LABEL: Testing : dynamic_broadcast_in_dim
func @dynamic_broadcast_in_dim(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<2xindex> {
  %0 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK:      Value info for %2 = shape.shape_of %1 : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  %2 = shape.shape_of %1 : tensor<?x?xf32> -> tensor<2xindex>
  return %2 : tensor<2xindex>
}

// -----

// CHECK-LABEL: Testing : dynamic_reshape
func @dynamic_reshape(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<2xindex> {
  %0 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %1 = "mhlo.dynamic_reshape"(%arg0, %0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK:      Value info for %2 = shape.shape_of %1 : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  %2 = shape.shape_of %1 : tensor<?x?xf32> -> tensor<2xindex>
  return %2 : tensor<2xindex>
}

// -----

// CHECK-LABEL: Testing : reduce
func @reduce(%arg0: tensor<?x?x?xf32>, %arg1: tensor<f32>) -> tensor<2xindex> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):  // no predecessors
    %26 = mhlo.add %a, %b : tensor<f32>
    "mhlo.return"(%26) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK:      Value info for %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?x?xf32>' at index: 0)[0]
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?x?xf32>' at index: 0)[2]
  %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: Testing : transpose
func @transpose(%arg0: tensor<?x?xf32>) -> tensor<2xindex> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK:      Value info for %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: Testing : select
func @select(%arg0: tensor<i1>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<2xindex> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2)  : (tensor<i1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK:      Value info for %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[0]
  // CHECK-NEXT: s0 with
  // CHECK-NEXT:   s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[1]
  %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: Testing : dim
func @dim(%arg0: tensor<?x?xf32>) -> tensor<2xindex> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %t = tensor.from_elements %d0, %d0 : tensor<2xindex>
  // CHECK:      Value info for %1 = tensor.from_elements %0, %0 : tensor<2xindex>
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  return %t : tensor<2xindex>
}

// -----

// CHECK-LABEL: Testing : extract
func @extract(%arg0: tensor<?x?xf32>) -> tensor<2xindex> {
  %shape = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %c1 = arith.constant 1 : index
  %d0 = tensor.extract %shape[%c1] : tensor<2xindex>
  // CHECK:      Value info for %2 = tensor.from_elements %1, %1 : tensor<2xindex>
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  %t = tensor.from_elements %d0, %d0 : tensor<2xindex>
  return %t : tensor<2xindex>
}

// -----

// CHECK-LABEL: Testing : symbolic_constraint
func @symbolic_constraint(
  %arg0: tensor<?x?xf32>
    {cpurt.symbolic_shape = dense<[-3, -2]> : tensor<2xi64>},
  %arg1: tensor<?x?xf32>
    {cpurt.symbolic_shape = dense<[-4, -2]> : tensor<2xi64>}
) -> tensor<2xi32> {
  %0 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?x?xf32> -> tensor<2xindex>
  %2 = arith.index_cast %0 : tensor<2xindex> to tensor<2xi32>
  %3 = arith.index_cast %1 : tensor<2xindex> to tensor<2xi32>
  // CHECK:      Value info for %4 = mhlo.add %2, %3 : tensor<2xi32>:
  // CHECK-NEXT:   s0 + s1 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  // CHECK-NEXT:     s1 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 1)[0]
  // CHECK-NEXT:   s0 + s1 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  // CHECK-NEXT:     s1 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
  %4 = mhlo.add %2, %3 : tensor<2xi32>
  return %4 : tensor<2xi32>
}

// -----

// CHECK-LABEL: Testing : dynamic_reshape
func @dynamic_reshape(%arg0: tensor<?x8x?x64xf32>, %arg1: tensor<4xi32>)
    -> tensor<?x8x?x64xf32> {
  %0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %1 = shape.num_elements %0 : tensor<4xindex> -> index
  %2 = mhlo.compute_reshape_shape %1, %arg1 : index, tensor<4xi32>
      -> tensor<4xi32>
  // CHECK:      Shape info for %3 = "mhlo.dynamic_reshape"(%arg0, %2) : (tensor<?x8x?x64xf32>, tensor<4xi32>) -> tensor<?x8x?x64xf32>
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = %2 = mhlo.compute_reshape_shape %1, %arg1 : index, tensor<4xi32> -> tensor<4xi32>[0]
  // CHECK-NEXT:   8
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = %2 = mhlo.compute_reshape_shape %1, %arg1 : index, tensor<4xi32> -> tensor<4xi32>[2]
  // CHECK-NEXT:   64
  %3 = "mhlo.dynamic_reshape"(%arg0, %2)
      : (tensor<?x8x?x64xf32>, tensor<4xi32>) -> tensor<?x8x?x64xf32>
  return %3 : tensor<?x8x?x64xf32>
}

// -----

// Larger examples.

// CHECK-LABEL: Testing : softmax
func @softmax(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = mhlo.constant dense<-1> : tensor<1xi64>
  %1 = "mhlo.convert"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %3 = "mhlo.reduce"(%1, %2) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %26 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%26) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  %4 = "mhlo.convert"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %cst = arith.constant dense<1> : tensor<1xi32>
  // CHECK:      Value info for %5 = shape.shape_of
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  %5 = shape.shape_of %4 : tensor<?xf32> -> tensor<1xindex>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK:      Value info for %6 = tensor.extract
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  %6 = tensor.extract %5[%c0] : tensor<1xindex>
  // CHECK:      Value info for %7 = tensor.from_elements
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
  // CHECK-NEXT:   1
  %7 = tensor.from_elements %6, %c1 : tensor<2xindex>
  %8 = "mhlo.dynamic_reshape"(%4, %7) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x1xf32>
  %9 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %10 = shape.shape_of %8 : tensor<?x1xf32> -> tensor<2xindex>
  %11 = shape.cstr_broadcastable %9, %10 : tensor<2xindex>, tensor<2xindex>
  %12 = shape.assuming %11 -> (tensor<?x?xf32>) {
    // CHECK:      Value info for %26 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>:
    // CHECK-NEXT:   s0 with
    // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
    // CHECK-NEXT:   s0 with
    // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[1]
    %26 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
    // CHECK:      Value info for %27 = shape.shape_of
    // CHECK-NEXT:   s0 with
    // CHECK-NEXT:     s0 = shapeof(<block argument> of type 'tensor<?x?xf32>' at index: 0)[0]
    // CHECK-NEXT:   1
    %27 = shape.shape_of %8 : tensor<?x1xf32> -> tensor<2xindex>
    %28 = shape.broadcast %26, %27 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
    %29 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %28) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    %30 = "mhlo.dynamic_broadcast_in_dim"(%8, %28) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    %31 = mhlo.subtract %29, %30 : tensor<?x?xf32>
    shape.assuming_yield %31 : tensor<?x?xf32>
  }
  %13 = "mhlo.exponential"(%12) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = "mhlo.convert"(%13) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %16 = "mhlo.reduce"(%14, %15) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %26 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%26) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  %17 = "mhlo.convert"(%16) : (tensor<?xf32>) -> tensor<?xf32>
  %cst_0 = arith.constant dense<1> : tensor<1xi32>
  %18 = shape.shape_of %17 : tensor<?xf32> -> tensor<1xindex>
  %c1_1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %19 = tensor.extract %18[%c0_2] : tensor<1xindex>
  %20 = tensor.from_elements %19, %c1_1 : tensor<2xindex>
  %21 = "mhlo.dynamic_reshape"(%17, %20) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x1xf32>
  %22 = shape.shape_of %13 : tensor<?x?xf32> -> tensor<2xindex>
  %23 = shape.shape_of %21 : tensor<?x1xf32> -> tensor<2xindex>
  %24 = shape.cstr_broadcastable %22, %23 : tensor<2xindex>, tensor<2xindex>
  %25 = shape.assuming %24 -> (tensor<?x?xf32>) {
    %26 = shape.shape_of %13 : tensor<?x?xf32> -> tensor<2xindex>
    %27 = shape.shape_of %21 : tensor<?x1xf32> -> tensor<2xindex>
    %28 = shape.broadcast %26, %27 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
    %29 = "mhlo.dynamic_broadcast_in_dim"(%13, %28) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    %30 = "mhlo.dynamic_broadcast_in_dim"(%21, %28) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    %31 = mhlo.divide %29, %30 : tensor<?x?xf32>
    shape.assuming_yield %31 : tensor<?x?xf32>
  }
  return %25 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: Testing : reshape_integration
func @reshape_integration(%arg0: tensor<512x512xf32>, %arg1: tensor<?x8x?x64xf32>, %arg2: tensor<4xi32>, %arg3: tensor<512xf32>, %arg4: tensor<?x?x512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512x2048xf32>, %arg8: tensor<2048xf32>, %arg9: tensor<2048x512xf32>, %arg10: tensor<512xf32>, %arg11: tensor<512xf32>, %arg12: tensor<512xf32>) -> tensor<?x512xf32> {
  %0 = mhlo.constant dense<512> : tensor<1xi32>
  %1 = shape.shape_of %arg1 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %2 = shape.num_elements %1 : tensor<4xindex> -> index
  %3 = mhlo.cstr_reshapable %2, %arg2 : index, tensor<4xi32>
  %4 = "mhlo.dynamic_reshape"(%arg1, %arg2) : (tensor<?x8x?x64xf32>, tensor<4xi32>) -> tensor<?x8x?x64xf32>
  %5 = "mhlo.transpose"(%4) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<?x8x?x64xf32>) -> tensor<?x?x8x64xf32>
  %6 = "mhlo.transpose"(%5) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<?x?x8x64xf32>) -> tensor<?x?x64x8xf32>
  %7 = shape.shape_of %6 : tensor<?x?x64x8xf32> -> tensor<4xindex>
  %8 = arith.index_cast %7 : tensor<4xindex> to tensor<4xi32>
  %9 = "mhlo.slice"(%8) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1xi32>
  %10 = "mhlo.reshape"(%9) : (tensor<1xi32>) -> tensor<i32>
  %11 = "mhlo.slice"(%8) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1xi32>
  %12 = "mhlo.reshape"(%11) : (tensor<1xi32>) -> tensor<i32>
  %13 = mhlo.multiply %10, %12 : tensor<i32>
  %14 = "mhlo.reshape"(%13) : (tensor<i32>) -> tensor<1xi32>
  // CHECK:      Value info for %15 = "mhlo.concatenate"(%14, %0) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT:   s0 * s1 with
  // CHECK-NEXT:     s0 = <block argument> of type 'tensor<4xi32>' at index: 2[0]
  // CHECK-NEXT:     s1 = <block argument> of type 'tensor<4xi32>' at index: 2[2]
  // CHECK-NEXT:   512
  %15 = "mhlo.concatenate"(%14, %0) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK:      Value info for %16 = shape.shape_of %6 : tensor<?x?x64x8xf32> -> tensor<4xindex>:
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = <block argument> of type 'tensor<4xi32>' at index: 2[0]
  // CHECK-NEXT:   s0 with
  // CHECK-NEXT:     s0 = <block argument> of type 'tensor<4xi32>' at index: 2[2]
  // CHECK-NEXT:   64
  // CHECK-NEXT:   8
  %16 = shape.shape_of %6 : tensor<?x?x64x8xf32> -> tensor<4xindex>
  %17 = shape.num_elements %16 : tensor<4xindex> -> index
  %18 = mhlo.cstr_reshapable %17, %15 : index, tensor<2xi32>
  %19 = shape.assuming %18 -> (tensor<?x512xf32>) {
    %21 = "mhlo.dynamic_reshape"(%6, %15) : (tensor<?x?x64x8xf32>, tensor<2xi32>) -> tensor<?x512xf32>
    shape.assuming_yield %21 : tensor<?x512xf32>
  }
  return %19 : tensor<?x512xf32>
}
