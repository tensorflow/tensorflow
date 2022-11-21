// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=: -tf-xla-emit-return-tuple | FileCheck %s


// TODO(b/259459405): Remove this test along with the upstream refactoring to
// avoid non TF inputs.
// This is not a supported mode.
module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }
}

// CHECK-LABEL: HloModule main
// CHECK:       ENTRY %main.{{[0-9]+}} ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> (f32[]) {
// CHECK-NEXT:    %[[ARG0]] = f32[] parameter(0)
// CHECK-NEXT:    %[[ARG1]] = f32[] parameter(1)
// CHECK-NEXT:    [[ADD:%.*]] = f32[] add(f32[] %[[ARG0]], f32[] %[[ARG1]])
// CHECK-NEXT:    ROOT %tuple.{{[0-9]+}} = (f32[]) tuple(f32[] [[ADD]])
// CHECK-NEXT:  }
