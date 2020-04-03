// RUN: tf-opt -tensor-linalg-to-buffer-linalg --buffer-assignment -split-input-file %s | FileCheck %s -dump-input-on-failure

#map0 = affine_map<(d0) -> (d0)>

module {
    // CHECK-LABEL: func @muliple_results_generic_op
    func @muliple_results_generic_op(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
        %0, %1 = linalg.generic {args_in = 1 : i64, args_out = 2 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} %arg0 {
        ^bb0(%arg1: f32):
            %1 = exp %arg1 : f32
            linalg.yield %1, %1 : f32, f32
        }: tensor<4xf32> -> (tensor<4xf32>, tensor<4xf32>)
        return %0, %1 : tensor<4xf32>, tensor<4xf32>
    }
}
//      CHECK: (%[[NEW_ARG0:.*]]: [[TYPE:.*]], %[[ARG1_RESULT:.*]]: [[TYPE]], %[[ARG2_RESULT:.*]]: [[TYPE]])
//      CHECK: %[[FIRST_ALLOC:.*]] = alloc() : [[TYPE]]
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc() : [[TYPE]]
//      CHECK: linalg.generic
// CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}
//      CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %[[ARG0:.*]]: f32, %{{.*}}: f32, %{{.*}}: f32
//      CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = exp %[[ARG0]]
//      CHECK: linalg.yield %[[RESULT]], %[[RESULT]]
//      CHECK: [[TYPE]], [[TYPE]], [[TYPE]]
// CHECK-NEXT: linalg.copy(%[[FIRST_ALLOC]], %[[ARG1_RESULT]])
// CHECK-NEXT: dealloc %[[FIRST_ALLOC]]
// CHECK-NEXT: linalg.copy(%[[SECOND_ALLOC]], %[[ARG2_RESULT]])
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return