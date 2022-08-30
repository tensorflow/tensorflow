// RUN: mlir-hlo-opt -hlo-legalize-sort -canonicalize %s | FileCheck %s

func.func @sort(%arg0 : tensor<37xi32>, %arg1 : tensor<37xi32>) -> (tensor<37xi32>, tensor<37xi32>) {
  %result:2 = "mhlo.sort"(%arg0, %arg1) ({
    ^bb0(%00: tensor<i32>, %01: tensor<i32>, %10: tensor<i32>, %11: tensor<i32>):
      %50 = tensor.extract %00[] : tensor<i32>
      %51 = tensor.extract %01[] : tensor<i32>
      %52 = arith.cmpi sgt, %50, %51 : i32
      %cmp_result = tensor.from_elements %52 : tensor<i1>
      "mhlo.return"(%cmp_result) : (tensor<i1>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<37xi32>, tensor<37xi32>) -> (tensor<37xi32>, tensor<37xi32>)
  func.return %result#0, %result#1 : tensor<37xi32>, tensor<37xi32>
}
// CHECK: func.func @sort(%[[ARG0:.*]]: tensor<37xi32>, %[[ARG1:.*]]: tensor<37xi32>) -> (tensor<37xi32>, tensor<37xi32>) {
// CHECK:   %[[TRUE:.*]] = arith.constant true
// CHECK:   %[[C16:.*]] = arith.constant 16 : index
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[C37:.*]] = arith.constant 37 : index
// CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<37xi32>
// CHECK:   %[[ALLOC_0:.*]] = memref.alloc() : memref<37xi32>
// CHECK:   %[[ALLOC_1:.*]] = memref.alloc() : memref<37xi32>
// CHECK:   %[[ALLOC_2:.*]] = memref.alloc() : memref<37xi32>
// CHECK:   scf.for %[[ARG2:.*]] = %[[C0]] to %[[C37]] step %[[C16]] {
// CHECK:     %[[ADDI:.*]] = arith.addi %[[ARG2]], %[[C16]] : index
// CHECK:     %[[MINSI:.*]] = arith.minsi %[[ADDI]], %[[C37]] : index
// CHECK:     %[[EXTRACT:.*]] = tensor.extract %[[ARG0]][%[[ARG2]]] : tensor<37xi32>
// CHECK:     %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]][%[[ARG2]]] : tensor<37xi32>
// CHECK:     memref.store %[[EXTRACT]], %[[ALLOC]][%[[ARG2]]] : memref<37xi32>
// CHECK:     memref.store %[[EXTRACT_0]], %[[ALLOC_1]][%[[ARG2]]] : memref<37xi32>
// CHECK:     %[[ADDI_0:.*]] = arith.addi %[[ARG2]], %[[C1]] : index
// CHECK:     scf.for %[[ARG3:.*]] = %[[ADDI_0]] to %[[MINSI]] step %[[C1]] {
// CHECK:       %[[EXTRACT_1:.*]] = tensor.extract %[[ARG0]][%[[ARG3]]] : tensor<37xi32>
// CHECK:       %[[EXTRACT_2:.*]] = tensor.extract %[[ARG1]][%[[ARG3]]] : tensor<37xi32>
// CHECK:       %[[WHILE:.*]]:2 = scf.while (%[[ARG4:.*]] = %[[ARG2]], %[[ARG5:.*]] = %[[ARG3]]) : (index, index) -> (index, index) {
// CHECK:         %[[CMPI:.*]] = arith.cmpi slt, %[[ARG4]], %[[ARG5]] : index
// CHECK:         scf.condition(%[[CMPI]]) %[[ARG4]], %[[ARG5]] : index, index
// CHECK:       } do {
// CHECK:       ^bb0(%[[ARG4]]: index, %[[ARG5]]: index):
// CHECK:         %[[ADDI_1:.*]] = arith.addi %[[ARG4]], %[[ARG5]] : index
// CHECK:         %[[SHRUI:.*]] = arith.shrui %[[ADDI_1]], %[[C1]] : index
// CHECK:         %[[ADDI_2:.*]] = arith.addi %[[SHRUI]], %[[C1]] : index
// CHECK:         %[[LOAD:.*]] = memref.load %[[ALLOC]][%[[SHRUI]]] : memref<37xi32>
// CHECK:         %[[CMPI_0:.*]] = arith.cmpi sgt, %[[EXTRACT_1]], %[[LOAD]] : i32
// CHECK:         %[[SELECT:.*]] = arith.select %[[CMPI_0]], %[[ARG4]], %[[ADDI_2]] : index
// CHECK:         %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[SHRUI]], %[[ARG5]] : index
// CHECK:         scf.yield %[[SELECT]], %[[SELECT_0]] : index, index
// CHECK:       }
// CHECK:       %[[SUBI:.*]] = arith.subi %[[ARG3]], %[[WHILE]]#0 : index
// CHECK:       scf.for %[[ARG4]] = %[[C0]] to %[[SUBI]] step %[[C1]] {
// CHECK:         %[[SUBI_0:.*]] = arith.subi %[[ARG3]], %[[ARG4]] : index
// CHECK:         %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[C1]] : index
// CHECK:         %[[LOAD_0:.*]] = memref.load %[[ALLOC]][%[[SUBI_1]]] : memref<37xi32>
// CHECK:         %[[LOAD_1:.*]] = memref.load %[[ALLOC_1]][%[[SUBI_1]]] : memref<37xi32>
// CHECK:         memref.store %[[LOAD_0]], %[[ALLOC]][%[[SUBI_0]]] : memref<37xi32>
// CHECK:         memref.store %[[LOAD_1]], %[[ALLOC_1]][%[[SUBI_0]]] : memref<37xi32>
// CHECK:       }
// CHECK:       memref.store %[[EXTRACT_1]], %[[ALLOC]][%[[WHILE]]#0] : memref<37xi32>
// CHECK:       memref.store %[[EXTRACT_2]], %[[ALLOC_1]][%[[WHILE]]#0] : memref<37xi32>
// CHECK:     }
// CHECK:   }
// CHECK:   %[[WHILE_0:.*]]:6 = scf.while (%[[ARG2]] = %[[C16]], %[[ARG3_0:.*]] = %[[FALSE]], %[[ARG4_0:.*]] = %[[ALLOC]], %[[ARG5_0:.*]] = %[[ALLOC_1]], %[[ARG6:.*]] = %[[ALLOC_0]], %[[ARG7:.*]] = %[[ALLOC_2]])
// CHECK:     %[[CMPI_1:.*]] = arith.cmpi slt, %[[ARG2]], %[[C37]] : index
// CHECK:     scf.condition(%[[CMPI_1]]) %[[ARG2]], %[[ARG3_0]], %[[ARG4_0]], %[[ARG5_0]], %[[ARG6]], %[[ARG7]]
// CHECK:   } do {
// CHECK:   ^bb0(%[[ARG2]]: index, %[[ARG3_0]]: i1, %[[ARG4_0]]: memref<37xi32>, %[[ARG5_0]]: memref<37xi32>, %[[ARG6]]: memref<37xi32>, %[[ARG7]]: memref<37xi32>):
// CHECK:     %[[ADDI_3:.*]] = arith.addi %[[ARG2]], %[[ARG2]] : index
// CHECK:     scf.for %[[ARG8:.*]] = %[[C0]] to %[[C37]] step %[[ADDI_3]] {
// CHECK:       %[[ADDI_4:.*]] = arith.addi %[[ARG8]], %[[ARG2]] : index
// CHECK:       %[[MINSI_0:.*]] = arith.minsi %[[ADDI_4]], %[[C37]] : index
// CHECK:       %[[ADDI_5:.*]] = arith.addi %[[ARG8]], %[[ADDI_3]] : index
// CHECK:       %[[MINSI_1:.*]] = arith.minsi %[[ADDI_5]], %[[C37]] : index
// CHECK:       %[[WHILE_1:.*]]:3 = scf.while (%[[ARG9:.*]] = %[[ARG8]], %[[ARG10:.*]] = %[[ARG8]], %[[ARG11:.*]] = %[[MINSI_0]]) : (index, index, index) -> (index, index, index) {
// CHECK:         %[[CMPI_2:.*]] = arith.cmpi slt, %[[ARG10]], %[[MINSI_0]] : index
// CHECK:         %[[CMPI_3:.*]] = arith.cmpi slt, %[[ARG11]], %[[MINSI_1]] : index
// CHECK:         %[[ANDI:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:         scf.condition(%[[ANDI]]) %[[ARG9]], %[[ARG10]], %[[ARG11]] : index, index, index
// CHECK:       } do {
// CHECK:         ^bb0(%[[ARG9]]: index, %[[ARG10]]: index, %[[ARG11]]: index):
// CHECK:         %[[LOAD_2:.*]] = memref.load %[[ARG4_0]][%[[ARG10]]] : memref<37xi32>
// CHECK:         %[[LOAD_3:.*]] = memref.load %[[ARG5_0]][%[[ARG10]]] : memref<37xi32>
// CHECK:         %[[LOAD_4:.*]] = memref.load %[[ARG4_0]][%[[ARG11]]] : memref<37xi32>
// CHECK:         %[[LOAD_5:.*]] = memref.load %[[ARG5_0]][%[[ARG11]]] : memref<37xi32>
// CHECK:         %[[CMPI_4:.*]] = arith.cmpi sgt, %[[LOAD_4]], %[[LOAD_2]] : i32
// CHECK:         %[[SELECT_1:.*]] = arith.select %[[CMPI_4]], %[[LOAD_4]], %[[LOAD_2]] : i32
// CHECK:         %[[SELECT_2:.*]] = arith.select %[[CMPI_4]], %[[LOAD_5]], %[[LOAD_3]] : i32
// CHECK:         memref.store %[[SELECT_1]], %[[ARG6]][%[[ARG9]]] : memref<37xi32>
// CHECK:         memref.store %[[SELECT_2]], %[[ARG7]][%[[ARG9]]] : memref<37xi32>
// CHECK:         %[[ADDI_6:.*]] = arith.addi %[[ARG10]], %[[C1]] : index
// CHECK:         %[[SELECT_3:.*]] = arith.select %[[CMPI_4]], %[[ARG10]], %[[ADDI_6]] : index
// CHECK:         %[[ADDI_7:.*]] = arith.addi %[[ARG11]], %[[C1]] : index
// CHECK:         %[[SELECT_4:.*]] = arith.select %[[CMPI_4]], %[[ADDI_7]], %[[ARG11]] : index
// CHECK:         %[[ADDI_8:.*]] = arith.addi %[[ARG9]], %[[C1]] : index
// CHECK:         scf.yield %[[ADDI_8]], %[[SELECT_3]], %[[SELECT_4]] : index, index, index
// CHECK:       }
// CHECK:       %[[CMPI_5:.*]] = arith.cmpi slt, %[[WHILE_1]]#1, %[[MINSI_0]] : index
// CHECK:       %[[SELECT_5:.*]] = arith.select %[[CMPI_5]], %[[WHILE_1]]#1, %[[WHILE_1]]#2 : index
// CHECK:       %[[SELECT_6:.*]] = arith.select %[[CMPI_5]], %[[MINSI_0]], %[[MINSI_1]] : index
// CHECK:       %[[SUBI_2:.*]] = arith.subi %[[SELECT_6]], %[[SELECT_5]] : index
// CHECK:       scf.for %[[ARG9]] = %[[C0]] to %[[SUBI_2]] step %[[C1]] {
// CHECK:       %[[ADDI_9:.*]] = arith.addi %[[SELECT_5]], %[[ARG9]] : index
// CHECK:       %[[ADDI_10:.*]] = arith.addi %[[WHILE_1]]#0, %[[ARG9]] : index
// CHECK:       %[[LOAD_6:.*]] = memref.load %[[ARG4_0]][%[[ADDI_9]]] : memref<37xi32>
// CHECK:       %[[LOAD_7:.*]] = memref.load %[[ARG5_0]][%[[ADDI_9]]] : memref<37xi32>
// CHECK:       memref.store %[[LOAD_6]], %[[ARG6]][%[[ADDI_10]]] : memref<37xi32>
// CHECK:       memref.store %[[LOAD_7]], %[[ARG7]][%[[ADDI_10]]] : memref<37xi32>
// CHECK:       }
// CHECK:     }
// CHECK:     %[[SUBI_3:.*]] = arith.subi %[[TRUE]], %[[ARG3_0]] : i1
// CHECK:     scf.yield %[[ADDI_3]], %[[SUBI_3]], %[[ARG6]], %[[ARG7]], %[[ARG4_0]], %[[ARG5_0]] : index, i1, memref<37xi32>, memref<37xi32>, memref<37xi32>, memref<37xi32>
// CHECK:   }
// CHECK:   %[[SELECT_7:.*]] = arith.select %[[WHILE_0]]#1, %[[ALLOC_0]], %[[ALLOC]] : memref<37xi32>
// CHECK:   %[[TO:.*]] = bufferization.to_tensor %[[SELECT_7]] : memref<37xi32>
// CHECK:   %[[SELECT_8:.*]] = arith.select %[[WHILE_0]]#1, %[[ALLOC_2]], %[[ALLOC_1]] : memref<37xi32>
// CHECK:   %[[TO_0:.*]] = bufferization.to_tensor %[[SELECT_8]] : memref<37xi32>
// CHECK:   return %[[TO]], %[[TO_0]] : tensor<37xi32>, tensor<37xi32>
