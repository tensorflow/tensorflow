// RUN: mlir-opt %s -mlir-elide-elementsattrs-if-larger=2 | FileCheck %s

// CHECK: dense<...> : tensor<3xi32>
"test.dense_attr"() {foo.dense_attr = dense<[1, 2, 3]> : tensor<3xi32>} : () -> ()

// CHECK: dense<[1, 2]> : tensor<2xi32>
"test.non_elided_dense_attr"() {foo.dense_attr = dense<[1, 2]> : tensor<2xi32>} : () -> ()

// CHECK: sparse<..., -2.{{0+}}e+00> : vector<1x1x1xf16>
"test.sparse_attr"() {foo.sparse_attr = sparse<[[1, 2, 3]],  -2.0> : vector<1x1x1xf16>} : () -> ()
