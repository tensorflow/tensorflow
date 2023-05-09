// RUN: mlir-hlo-opt %s --split-input-file --computeop-and-func-bufferize \
// RUN:     --allow-unregistered-dialect --final-bufferize=alignment=128 | \
// RUN: FileCheck %s

func.func @sort(%input1: tensor<?x?x?xf32>, %input2: tensor<?x?x?xi32>,
                %init1: tensor<?x?x?xf32>, %init2: tensor<?x?x?xi32>)
    -> (tensor<?x?x?xf32>, tensor<?x?x?xi32>) {
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?x?xf32>, %input2: tensor<?x?x?xi32>)
      outs(%init1: tensor<?x?x?xf32>, %init2: tensor<?x?x?xi32>)
      dimension = 1
      is_stable = true
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?x?xf32>, tensor<?x?x?xi32>
}

// CHECK-LABEL:  func.func @sort
// CHECK-SAME:          (%[[INPUT1:[A-Za-z_0-9]*]]: memref<?x?x?xf32>,
// CHECK-SAME:           %[[INPUT2:[A-Za-z_0-9]*]]: memref<?x?x?xi32>,
// CHECK-SAME:           %[[INIT1:[A-Za-z_0-9]*]]: memref<?x?x?xf32>,
// CHECK-SAME:           %[[INIT2:[A-Za-z_0-9]*]]: memref<?x?x?xi32>)
// CHECK-SAME:       -> (memref<?x?x?xf32>, memref<?x?x?xi32>)
// CHECK-DAG:      %[[OUTPUT1:.*]] = memref.alloc
// CHECK-DAG:      memref.copy %[[INIT1]], %[[OUTPUT1]]
// CHECK-DAG:      %[[OUTPUT2:.*]] = memref.alloc
// CHECK-DAG:      memref.copy %[[INIT2]], %[[OUTPUT2]]
// CHECK:          thlo.sort
// CHECK-SAME:         ins(%[[INPUT1]] : memref<?x?x?xf32>,
// CHECK-SAME:           %[[INPUT2]] : memref<?x?x?xi32>)
// CHECK-SAME:         outs(%[[OUTPUT1]] : memref<?x?x?xf32>,
// CHECK-SAME:           %[[OUTPUT2]] : memref<?x?x?xi32>)
// CHECK-SAME:         dimension = 1
// CHECK-SAME:         is_stable = true
// CHECK-NEXT:         (%[[FLOAT1:[A-Za-z_0-9]*]]: f32, %[[FLOAT2:.*]]: f32,
// CHECK-SAME:          %[[INT1:[A-Za-z_0-9]*]]: i32, %[[INT2:.*]]: i32)
// CHECK:                 %[[RESULT:.*]] = arith.cmpf ogt, %[[FLOAT1]], %[[FLOAT2]] : f32
// CHECK:                 thlo.yield %[[RESULT]] : i1
// CHECK:          return %[[OUTPUT1]], %[[OUTPUT2]]
