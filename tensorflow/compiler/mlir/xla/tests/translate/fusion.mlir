// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK: %[[REGION0:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK: %[[REGION1:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> (f32[], f32[])
//
// CHECK: ENTRY
// CHECK:   %[[PARAM0:.*]] = f32[] parameter(0)
// CHECK:   %[[PARAM1:.*]] = f32[] parameter(1)
// CHECK:   %[[FUSION0:.*]] = f32[] fusion(f32[] %[[PARAM0]], f32[] %[[PARAM1]]), kind=kLoop, calls=%[[REGION0]]
// CHECK:   %[[FUSION1:.*]] = (f32[], f32[]) fusion(f32[] %[[PARAM0]], f32[] %[[PARAM1]]), kind=kLoop, calls=%[[REGION1]]
// CHECK:   f32[] get-tuple-element((f32[], f32[]) %[[FUSION1]]), index=0
// CHECK:   f32[] get-tuple-element((f32[], f32[]) %[[FUSION1]]), index=1
// CHECK: }
func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  %result = "mhlo.fusion"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %result = "mhlo.add"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%result) : (tensor<f32>) -> ()
    }) { fusion_kind = "kLoop" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %result0, %result1 = "mhlo.fusion"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %elem0 = "mhlo.add"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      %elem1 = "mhlo.subtract"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%elem0, %elem1) : (tensor<f32>, tensor<f32>) -> ()
    }) { fusion_kind="kLoop" } : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  return
}
