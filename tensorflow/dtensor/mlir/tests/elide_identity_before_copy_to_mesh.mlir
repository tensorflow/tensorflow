// RUN: dtensor-opt %s -split-input-file -dtensor-elide-identity-before-copy-to-mesh | FileCheck %s

// Check that identity before CopyToMeshGrad is elided.
// CHECK-LABEL: func @check_elide_identity
func.func @check_elide_identity() -> (tensor<4xi32>) {
    // CHECK: %[[CONST:.*]] = "tf.Const"()
    // CHECK-NEXT: %[[CONST_1:.*]] = "tf.Const"()
    // CHECK-NEXT: "tf.CopyToMeshGrad"(%[[CONST]], %[[CONST_1]])

    %cst = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
    %cst_1 = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
    %1 = "tf.Identity"(%cst) : (tensor<4xi32>) -> tensor<4xi32>
    %2 = "tf.CopyToMeshGrad"(%1, %cst_1) {reference_layout=""}: (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    func.return %2 : tensor<4xi32>
}


