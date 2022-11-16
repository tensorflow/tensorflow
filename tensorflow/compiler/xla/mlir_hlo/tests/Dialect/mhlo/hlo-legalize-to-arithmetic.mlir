// RUN: mlir-hlo-opt -hlo-legalize-to-arithmetic %s | FileCheck %s

func.func @reshape_unsigned() -> tensor<2xui64> {
  %result = mhlo.xla.rng_get_and_update_state {delta = 1 : i64}
  func.return %result : tensor<2xui64>
}

// CHECK: memref.global "private" @rng_state : memref<i128>

// CHECK-LABEL:     func @reshape_unsigned
// CHECK: %[[GLOBAL:.*]] = memref.get_global @rng_state : memref<i128>
// CHECK: %[[OLD_SEED:.*]] = memref.load %[[GLOBAL]][] : memref<i128>
// CHECK: %[[DELTA:.*]] = arith.constant 1 : i128
// CHECK: %[[NEW_SEED:.*]] = arith.addi %[[OLD_SEED]], %[[DELTA]] : i128
// CHECK: memref.store %[[NEW_SEED]], %[[GLOBAL]][] : memref<i128>
// CHECK: %[[C64:.*]] = arith.constant 64 : i128
// CHECK: %[[UPPER_BITS:.*]] = arith.shrui %[[OLD_SEED]], %[[C64]] : i128
// CHECK: %[[UPPER_WORD:.*]] = arith.trunci %[[UPPER_BITS]] : i128 to i64
// CHECK: %[[C0:.*]] = arith.constant 0 : i128
// CHECK: %[[LOWER_BITS:.*]] = arith.shrui %[[OLD_SEED]], %[[C0]] : i128
// CHECK: %[[LOWER_WORD:.*]] = arith.trunci %[[LOWER_BITS]] : i128 to i64

// CHECK: %[[PACKED:.*]] = tensor.from_elements %[[UPPER_WORD]],
// CHECK-SAME:  %[[LOWER_WORD]] : tensor<2xi64>

// CHECK: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[PACKED]]
// CHECK-SAME: tensor<2xi64> to tensor<2xui64>
// CHECK: return %[[CASTED_RESULT]] : tensor<2xui64>
