// RUN: tf-opt -tensor-linalg-to-buffer-linalg -split-input-file %s | FileCheck %s -dump-input-on-failure

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {
    // CHECK-LABEL: func @func_exp
    func @func_exp(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %0 = linalg.generic {args_in = 1 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} %arg0 {
        ^bb0(%arg1: f32):   // no predecessors
            %1 = exp %arg1 : f32
            linalg.yield %1 : f32
        }: tensor<2x2xf32> -> tensor<2x2xf32>
        return %0 : tensor<2x2xf32>
    }
}
//      CHECK: (%[[NEW_ARG0:.*]]: [[TYPE:.*]], %[[ARG_RESULT:.*]]: [[TYPE]])
//      CHECK: %[[ALLOC:.*]] = alloc() : [[TYPE]]
//      CHECK: linalg.generic
// CHECK-SAME: %{{.*}}, %{{.*}}
//      CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %[[ARG0:.*]]: f32, %{{.*}}: f32
//      CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = exp %[[ARG0]]
//      CHECK: linalg.yield %[[RESULT]]
//      CHECK: [[TYPE]], [[TYPE]]
//      CHECK: linalg.copy(%[[ALLOC]], %[[ARG_RESULT]])
// CHECK-NEXT: return
