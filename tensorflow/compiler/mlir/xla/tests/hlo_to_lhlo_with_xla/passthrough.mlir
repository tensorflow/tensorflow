// RUN: xla-opt -xla-hlo-to-lhlo-with-xla %s | FILECHECK_OPTS="" FileCheck --enable-var-scope %s

// Current allocation will lead to one buffer argument for the "value" and
// another one for the output, an no returned values.
// CHECK-LABEL: func @main
// CHECK-SAME:  %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0 : index},
// CHECK-SAME:  %[[ARG1:.*]]: memref<16xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}
// CHECK-SAME: ) {{.*}} {
func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // The only expected instruction is a copy from the input into the output.
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[OUTPUT:.*]] = memref.view %[[ARG1]][%[[C0]]][] : memref<16xi8> to memref<2x2xf32>
  // CHECK: lmhlo.copy
  // CHECK-SAME: %[[ARG0]], %[[OUTPUT]]
  return %value : tensor<2x2xf32>
}
