// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s --dump-input=always --check-prefixes=CHECK


#loc1 = loc("x")
#loc2 = loc("mhlo.original_value={{\22x\22}}")
#loc3 = loc("y")
#loc4 = loc("mhlo.original_value={{\22y\22}}")
#loc5 = loc("add0")
#loc6 = loc("Add_0")
#loc7 = loc("mhlo.original_value={{\22add1\22}}")
#loc8 = loc("Add_1")
#loc9 = loc("source.txt":17:0)
#loc10 = loc("mhlo.original_value={{\22add2\22}}")
#loc11 = loc(fused[#loc1, #loc2])
#loc12 = loc(fused[#loc3, #loc4])
#loc13 = loc(fused[#loc6, #loc7])
#loc14 = loc(fused[#loc8, #loc9, #loc10])
#loc15 = loc("TopK_0")
#loc16 = loc("mhlo.original_value={({\22t\22 {0}}, {\22t\22 {1}})}")
#loc17 = loc(fused[#loc15, #loc16])

module @Test {

// CHECK-LABEL: main
// CHECK: %[[ARG0:.*]] = f32[4] parameter(0), origin={{[{][{]}}"x"{{[}][}]}}
// CHECK: %[[ARG1:.*]] = f32[4] parameter(1), origin={{[{][{]}}"y"{{[}][}]}}
// CHECK: %[[ADD0:.*]] = f32[4] add(%[[ARG0]], %[[ARG1]])
// CHECK-NOT: origin
// CHECK-SAME: metadata={op_name="add0"}
// CHECK: %[[ADD1:.*]] = f32[4] add(%[[ADD0]], %[[ARG1]]), origin={{[{][{]}}"add1"{{[}][}]}}, metadata=
// CHECK: ROOT %[[ADD2:.*]] = f32[4] add(%[[ADD1]], %[[ARG1]]), origin={{[{][{]}}"add2"{{[}][}]}}, metadata=
  func.func @main(%arg0: tensor<4xf32> loc(fused[#loc1, #loc2]), %arg1: tensor<4xf32> loc(fused[#loc3, #loc4])) -> tensor<4xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4xf32> loc(#loc5)
    %1 = mhlo.add %0, %arg1 : tensor<4xf32> loc(#loc13)
    %2 = mhlo.add %1, %arg1 : tensor<4xf32> loc(#loc14)
    return %2 : tensor<4xf32>
  }


// CHECK-LABEL: tuple_results
// CHECK:  %Arg_0.1 = f32[10] parameter(0)
// CHECK:  %[[TOPK:.*]] = (f32[8], s32[8]) topk(%Arg_0.1), k=8, largest=true, origin={({"t" {0}{{[}]}}, {"t" {1}{{[}]}})}
// CHECK:  ROOT %[[GTE0:.*]] = f32[8] get-tuple-element(%[[TOPK]]), index=0, origin={{[{][{]}}"t" {0}{{[}][}]}}
// CHECK:  %[[GTE1:.*]] = s32[8] get-tuple-element(%[[TOPK]]), index=1, origin={{[{][{]}}"t" {1}{{[}][}]}}
  func.func @tuple_results(%arg0: tensor<10xf32>) -> tensor<8xf32> {
    %0:2 = mhlo.topk(%arg0, k=8, largest=true) : tensor<10xf32> -> (tensor<8xf32>, tensor<8xi32>) loc(#loc17)
    return %0#0 : tensor<8xf32>
  }

}


