// RUN: mlir-hlo-opt %s --split-input-file --mhlo-restrict-max-rank | \
// RUN: FileCheck %s

// CHECK-LABEL: @ReduceTransposeReduce4D
func.func @ReduceTransposeReduce4D(%arg0 : tensor<17x6x35x13xf32>) -> tensor<357x2x5x13xf32> {

  // CHECK: %[[OUT0:.*]] = mhlo.reshape %arg0 : (tensor<17x6x35x13xf32>) -> tensor<17x6x5x7x13xf32>
  // CHECK: %[[OUT1:.*]] = "mhlo.transpose"(%[[OUT0]]) <{permutation = dense<[3, 0, 1, 2, 4]> : tensor<5xi64>}> : (tensor<17x6x5x7x13xf32>) -> tensor<7x17x6x5x13xf32>
  // CHECK: %[[OUT2:.*]] = mhlo.reshape %[[OUT1]] : (tensor<7x17x6x5x13xf32>) -> tensor<119x2x3x5x13xf32>
  // CHECK: %[[OUT3:.*]] = "mhlo.transpose"(%[[OUT2]]) <{permutation = dense<[2, 0, 1, 3, 4]> : tensor<5xi64>}> : (tensor<119x2x3x5x13xf32>) -> tensor<3x119x2x5x13xf32>
  // CHECK: %[[OUT4:.*]] = mhlo.reshape %[[OUT3]] : (tensor<3x119x2x5x13xf32>) -> tensor<357x2x5x13xf32>
  // CHECK: return %[[OUT4]]

  %0 = "mhlo.reshape"(%arg0) : (tensor<17x6x35x13xf32>) -> tensor<17x2x3x5x7x13xf32>
  %1 = "mhlo.transpose"(%0) <{permutation = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi64>}> : (tensor<17x2x3x5x7x13xf32>) -> tensor<3x7x17x2x5x13xf32>
  %2 = "mhlo.reshape"(%1) : (tensor<3x7x17x2x5x13xf32>) -> tensor<357x2x5x13xf32>
  return %2 : tensor<357x2x5x13xf32>
}

// -----

// CHECK-LABEL: @ReduceTransposeReduce5D
func.func @ReduceTransposeReduce5D(%arg0 : tensor<17x6x35x15x13xf32>) -> tensor<1785x2x5x3x13xf32> {

  // CHECK: %[[OUT0:.*]] = mhlo.reshape %arg0 : (tensor<17x6x35x15x13xf32>) -> tensor<17x6x35x3x5x13xf32>
  // CHECK: %[[OUT1:.*]] = "mhlo.transpose"(%[[OUT0]]) <{permutation = dense<[4, 0, 1, 2, 3, 5]> : tensor<6xi64>}> : (tensor<17x6x35x3x5x13xf32>) -> tensor<5x17x6x35x3x13xf32>
  // CHECK: %[[OUT2:.*]] = mhlo.reshape %[[OUT1]] : (tensor<5x17x6x35x3x13xf32>) -> tensor<85x6x5x7x3x13xf32>
  // CHECK: %[[OUT3:.*]] = "mhlo.transpose"(%[[OUT2]]) <{permutation = dense<[3, 0, 1, 2, 4, 5]> : tensor<6xi64>}> : (tensor<85x6x5x7x3x13xf32>) -> tensor<7x85x6x5x3x13xf32>
  // CHECK: %[[OUT4:.*]] = mhlo.reshape %[[OUT3]] : (tensor<7x85x6x5x3x13xf32>) -> tensor<595x2x3x5x3x13xf32>
  // CHECK: %[[OUT5:.*]] = "mhlo.transpose"(%[[OUT4]]) <{permutation = dense<[2, 0, 1, 3, 4, 5]> : tensor<6xi64>}> : (tensor<595x2x3x5x3x13xf32>) -> tensor<3x595x2x5x3x13xf32>
  // CHECK: %[[OUT6:.*]] = mhlo.reshape %[[OUT5]] : (tensor<3x595x2x5x3x13xf32>) -> tensor<1785x2x5x3x13xf32>
  // CHECK: return %[[OUT6]]

  %0 = "mhlo.reshape"(%arg0) : (tensor<17x6x35x15x13xf32>) -> tensor<17x2x3x5x7x3x5x13xf32>
  %1 = "mhlo.transpose"(%0) <{permutation = dense<[2, 4, 6, 0, 1, 3, 5, 7]> : tensor<8xi64>}> : (tensor<17x2x3x5x7x3x5x13xf32>) -> tensor<3x7x5x17x2x5x3x13xf32>
  %2 = "mhlo.reshape"(%1) : (tensor<3x7x5x17x2x5x3x13xf32>) -> tensor<1785x2x5x3x13xf32>
  return %2 : tensor<1785x2x5x3x13xf32>
}

// -----

// CHECK-LABEL: @ReduceTransposeReduce4D
func.func @ReduceTransposeReduce4D(%arg0 : tensor<17x6x35x13xf32>) -> tensor<357x2x5x13xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<17x6x35x13xf32>) -> tensor<17x2x3x5x7x13xf32>

  // Shouldn't modify this transpose op as it doesn't meet the criteria.
  // CHECK: "mhlo.transpose"(%{{.*}}) <{permutation = dense<[4, 2, 0, 1, 3, 5]> : tensor<6xi64>}> : (tensor<17x2x3x5x7x13xf32>) -> tensor<7x3x17x2x5x13xf32>

  %1 = "mhlo.transpose"(%0) <{permutation = dense<[4, 2, 0, 1, 3, 5]> : tensor<6xi64>}> : (tensor<17x2x3x5x7x13xf32>) -> tensor<7x3x17x2x5x13xf32>
  %2 = "mhlo.reshape"(%1) : (tensor<7x3x17x2x5x13xf32>) -> tensor<357x2x5x13xf32>
  return %2 : tensor<357x2x5x13xf32>
}

// -----

// CHECK-LABEL: @ReduceTransposeReduce4D
func.func @ReduceTransposeReduce4D(%arg0 : tensor<17x6x35x13xf32>) -> tensor<3x238x5x13xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<17x6x35x13xf32>) -> tensor<17x2x3x5x7x13xf32>

  // Shouldn't modify this transpose op as it doesn't meet the criteria.
  // CHECK: "mhlo.transpose"(%{{.*}}) <{permutation = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi64>}> : (tensor<17x2x3x5x7x13xf32>) -> tensor<3x7x17x2x5x13xf32>
  %1 = "mhlo.transpose"(%0) <{permutation = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi64>}> : (tensor<17x2x3x5x7x13xf32>) -> tensor<3x7x17x2x5x13xf32>

  %2 = "mhlo.reshape"(%1) : (tensor<3x7x17x2x5x13xf32>) -> tensor<3x238x5x13xf32>
  return %2 : tensor<3x238x5x13xf32>
}
