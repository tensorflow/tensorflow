// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=10,19:19,10 -tf-xla-emit-use-tuple-args -tf-xla-emit-return-tuple | FileCheck %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main(%arg0: tensor<10x19xf32>, %arg1: tensor<19x10xf32> {mhlo.is_same_data_across_replicas = true}) -> tensor<10x19xf32> {
    %0 = "tf.Shape"(%arg0) : (tensor<10x19xf32>) -> tensor<2xi64>
    %1 = "tf.Reshape"(%arg1, %0) : (tensor<19x10xf32>, tensor<2xi64>) -> tensor<10x19xf32>
    func.return %1 : tensor<10x19xf32>
  }
}

// Tests that foldable ops are constant-folded to enable legalization of ops
// that require compile time constant operand.
// "tf.Shape" can only be folded away after shape inference. tf.Reshape can only
// be lowered when tf.Shape is folded into a constant.

// CHECK-LABEL: HloModule main
// CHECK:       ENTRY %main.{{[0-9]+}} ([[ARG_TUPLE:.*]]: (f32[10,19], f32[19,10])) -> (f32[10,19]) {
// CHECK:         %[[ARG_TUPLE]] = (f32[10,19]{1,0}, f32[19,10]{1,0}) parameter(0), parameter_replication={false,true}
// CHECK:         [[ARG0:%.*]] = f32[10,19]{1,0} get-tuple-element((f32[10,19]{1,0}, f32[19,10]{1,0}) %[[ARG_TUPLE]]), index=0
// CHECK:         [[ARG1:%.*]] = f32[19,10]{1,0} get-tuple-element((f32[10,19]{1,0}, f32[19,10]{1,0}) %[[ARG_TUPLE]]), index=1
// CHECK:         [[RESHAPE:%.*]] = f32[10,19]{1,0} reshape(f32[19,10]{1,0} [[ARG1]])
// CHECK:         ROOT %tuple.{{[0-9]+}} = (f32[10,19]{1,0}) tuple(f32[10,19]{1,0} [[RESHAPE]])
// CHECK:       }
