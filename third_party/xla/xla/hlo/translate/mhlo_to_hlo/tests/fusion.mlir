// RUN: xla-translate -mlir-hlo-to-hlo-text -split-input-file %s | FileCheck %s

// CHECK: %[[REGION0:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK: %[[REGION1:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> (f32[], f32[])
// CHECK: %[[REGION2:.*]] ({{.*}}: f32[]) -> (f32[], f32[])
//
// CHECK: ENTRY
// CHECK:   %[[PARAM0:.*]] = f32[] parameter(0)
// CHECK:   %[[PARAM1:.*]] = f32[] parameter(1)
// CHECK:   %[[FUSION0:.*]] = f32[] fusion(%[[PARAM0]], %[[PARAM1]]), kind=kLoop, calls=%[[REGION0]]
// CHECK:   %[[FUSION1:.*]] = (f32[], f32[]) fusion(%[[PARAM0]], %[[PARAM1]]), kind=kLoop, calls=%[[REGION1]]
// CHECK:   f32[] get-tuple-element(%[[FUSION1]]), index=0
// CHECK:   f32[] get-tuple-element(%[[FUSION1]]), index=1
// CHECK:   %[[FUSION2:.*]] = (f32[], f32[]) fusion(%[[PARAM0]]), kind=kLoop, calls=%[[REGION2]]
// CHECK:   f32[] get-tuple-element(%[[FUSION2]]), index=0
// CHECK:   f32[] get-tuple-element(%[[FUSION2]]), index=1
// CHECK: }
func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  %result = "mhlo.fusion"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %result = "mhlo.add"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%result) : (tensor<f32>) -> ()
    }) { fusion_kind = #mhlo<fusion_kind kLoop>} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %result0, %result1 = "mhlo.fusion"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %elem0 = "mhlo.add"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      %elem1 = "mhlo.subtract"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%elem0, %elem1) : (tensor<f32>, tensor<f32>) -> ()
    }) { fusion_kind=#mhlo<fusion_kind kLoop> } : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %result2:2 = "mhlo.fusion"(%arg0) ( {
    ^bb0(%arg2: tensor<f32>):  // no predecessors
      %4 = mhlo.add %arg2, %arg2 : tensor<f32>
      %5 = mhlo.subtract %arg2, %arg2 : tensor<f32>
      "mhlo.return"(%4, %5) : (tensor<f32>, tensor<f32>) -> ()
    }) {fusion_kind = #mhlo<fusion_kind kLoop>} : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return
}

// -----
//
// CHECK{LITERAL}: output_to_operand_aliasing={{}: (0, {})}
func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  %result = "mhlo.fusion"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %result = "mhlo.add"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%result) : (tensor<f32>) -> ()
    }) { fusion_kind = #mhlo<fusion_kind kLoop>, output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>
    ] } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return
}