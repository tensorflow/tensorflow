// RUN: xla-opt -xla-hlo-to-lhlo-with-xla %s | FileCheck --enable-var-scope %s

// Current allocation will lead to one buffer argument for the "value" and
// another one for the output, an no returned values.
// CHECK-LABEL: func @main
// CHECK-SAME:  %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0 : index},
// CHECK-SAME:  %[[ARG1:.*]]: memref<16xi8> {xla_lhlo.alloc = 0 : index, xla_lhlo.liveout = true}
// CHECK-SAME: ) {
func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // The only expected instruction is a copy from the input into the output.
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[C02:.*]] = constant 0 : index
  // CHECK: %[[OUTPUT:.*]] = std.view %[[ARG1]][%[[C02]]][] : memref<16xi8> to memref<2x2xf32>
  // CHECK: xla_lhlo.copy
  // CHECK-SAME: %[[ARG0]], %[[OUTPUT]]
  return %value : tensor<2x2xf32>
}
