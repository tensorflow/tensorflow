// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-lower-tensors | FileCheck %s

module {
  func.func @add(%arg0: f32, %arg1: f32) -> f32 {
    %sum = arith.addf %arg0, %arg1 : f32
    func.return %sum : f32
  }

  func.func @tensorarg(%arg0: tensor<43xf32> {xla.invariant, xla.slice_index = 0}, %arg1: index) -> f32 {
    %v1 = arith.constant 2.0 : f32
    %v2 = tensor.extract %arg0[%arg1] : tensor<43xf32>
    %sum = func.call @add(%v1, %v2) : (f32, f32) -> f32
    func.return %sum : f32
  }

  func.func @tensorcall(%arg0: tensor<43xf32> {xla.slice_index = 0}, %arg1: index) -> f32 {
    %call = func.call @tensorarg(%arg0, %arg1) : (tensor<43xf32>, index) -> f32
    func.return %call : f32
  }

  func.func @stores(%arg0: tensor<17xf32> {xla.slice_index = 0}, %arg1: tensor<43xf32> {xla.slice_index = 1}) -> tensor<43xf32> {
    %c17 = arith.constant 17 : index
    %c23 = arith.constant 23 : index
    %cst = arith.constant 3.0 : f32
    %out = tensor.insert %cst into %arg1[%c17] : tensor<43xf32>
    %out2 = tensor.insert %cst into %out[%c23] : tensor<43xf32>
    func.return %out2 : tensor<43xf32>
  }
}

// CHECK:        func.func @add(%{{.*}}: f32, %{{.*}}: f32) -> f32 {
// CHECK-NEXT:     arith.addf
// CHECK-NEXT:     return

// CHECK:        func.func @tensorarg(%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:        {xla.invariant, xla.slice_index = 0 : i64}, %[[ARG1:.*]]: index) -> f32 {
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2.000000e+00
// CHECK-DAG:       %[[IDX:.*]] = arith.index_castui %[[ARG1]] : index to i64
// CHECK-DAG:       %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][%[[IDX]]]
// CHECK-DAG:       %[[V2:.*]] = llvm.load %[[PTR]] invariant
// CHECK:           %[[RET:.*]] = call @add(%[[C2]], %[[V2]])
// CHECK:           return %[[RET]]

// CHECK:        func.func @tensorcall(%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:        {xla.slice_index = 0 : i64}, %[[ARG1:.*]]: index)
// CHECK:           %[[RET:.*]] = call @tensorarg(%[[ARG0]], %[[ARG1]])
// CHECK:           return %[[RET]]

// CHECK:        func.func @stores(
// CHECK-SAME:        %[[ARG0:.*]]: !llvm.ptr {xla.slice_index = 0 : i64},
// CHECK-SAME:        %[[ARG1:.*]]: !llvm.ptr {xla.slice_index = 1 : i64})
// CHECK-NEXT:      %[[CST:.*]] = arith.constant 3.000000e+00 : f32
// CHECK-NEXT:      %[[PTR1:.*]] = llvm.getelementptr inbounds %[[ARG1]][17]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR1]]
// CHECK-NEXT:      %[[PTR2:.*]] = llvm.getelementptr inbounds %[[ARG1]][23]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR2]]
// CHECK-NEXT:      return

// -----

module {
  func.func @layout(
      %arg0: tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>,
      %arg1: index, %arg2: index) -> f32 {
    %v = tensor.extract %arg0[%arg1, %arg2]
        : tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>
    func.return %v : f32
  }
}

// CHECK:      #[[MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1 * 2)>
// CHECK:      @layout(%[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME:     %[[X:.*]]: index, %[[Y:.*]]: index
// CHECK:        %[[IDX:.*]] = affine.apply #[[MAP]](%[[X]], %[[Y]])
// CHECK:        %[[IDX_CAST:.*]] = arith.index_castui %[[IDX]]
// CHECK:        %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][%[[IDX_CAST]]]
// CHECK:        llvm.load %[[PTR]]

// -----

module {
  func.func @store_control_flow(
     %arg0: tensor<2xf32>,
     %arg1: index
  ) -> tensor<2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.0 : f32
    %cst2 = arith.constant 1.0 : f32

    %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
      %new_out = tensor.insert %cst into %arg2[%i] : tensor<2xf32>
      scf.yield %new_out : tensor<2xf32>
    }

    %inbounds = arith.cmpi sle, %arg1, %c1 : index
    %result = scf.if %inbounds -> tensor<2xf32> {
      %if = tensor.insert %cst2 into %for[%arg1] : tensor<2xf32>
      scf.yield %if : tensor<2xf32>
    } else {
      scf.yield %for : tensor<2xf32>
    }
    func.return %result : tensor<2xf32>
  }
}

// CHECK:     @store_control_flow(%[[ARG0:.*]]: !llvm.ptr, %[[X:.*]]: index) {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:         %[[CAST:.*]] = arith.index_castui %[[I]]
// CHECK:         %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][%[[CAST]]]
// CHECK:         llvm.store {{.*}}, %[[PTR]]
// CHECK:       %[[INBOUNDS:.*]] = arith.cmpi
// CHECK:       scf.if %[[INBOUNDS]] {
// CHECK:         llvm.store
// CHECK-NEXT:  }
// CHECK-NEXT:  return